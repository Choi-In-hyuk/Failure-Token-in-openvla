"""
run_failure_eval.py

Runs a model with failure detection in a LIBERO simulation environment.

Usage:
    python run_failure_eval.py \
        --model_checkpoint <PATH_TO_FAILURE_MODEL> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Dict

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# Import our failure detection model
from failure_model import OpenVLAWithFailureDetection


@dataclass
class FailureEvalConfig:
    # Model parameters
    model_checkpoint: Union[str, Path] = ""           # Path to our failure detection model checkpoint
    center_crop: bool = True                          # Center crop? (if trained w/ random crop image aug)
    failure_threshold: float = 0.5                    # Threshold for failure detection
    
    # LIBERO environment parameters
    task_suite_name: str = "libero_object"            # Task suite
    num_steps_wait: int = 10                          # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5                      # Number of rollouts per task
    
    # Utils
    run_id_note: Optional[str] = None                 # Extra note to add in run ID for logging
    local_log_dir: str = "./failure_eval_logs"       # Local directory for eval logs
    seed: int = 7                                     # Random Seed


class FailureMetricsTracker:
    """Tracks failure detection metrics across episodes"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_negative = 0   # Success episodes, no failure detected
        self.false_positive = 0  # Success episodes, but failure detected  
        self.false_negative = 0  # Failed episodes, but no failure detected
        self.true_positive = 0   # Failed episodes, failure detected
        
        self.episodes_with_failure_detection = 0
        self.total_failure_detections = 0
        self.failure_detection_history = []
        
    def update(self, episode_success: bool, failure_detected: bool, num_failure_detections: int = 0):
        """Update metrics for one episode"""
        if episode_success and not failure_detected:
            self.true_negative += 1
        elif episode_success and failure_detected:
            self.false_positive += 1
        elif not episode_success and not failure_detected:
            self.false_negative += 1
        elif not episode_success and failure_detected:
            self.true_positive += 1
            
        if failure_detected:
            self.episodes_with_failure_detection += 1
            
        self.total_failure_detections += num_failure_detections
        self.failure_detection_history.append({
            'success': episode_success,
            'failure_detected': failure_detected,
            'num_detections': num_failure_detections
        })
        
    def get_metrics(self):
        """Get confusion matrix and related metrics"""
        total = self.true_negative + self.false_positive + self.false_negative + self.true_positive
        
        if total == 0:
            return {}
            
        accuracy = (self.true_negative + self.true_positive) / total
        
        # Precision and Recall for failure detection
        precision_failure = self.true_positive / (self.true_positive + self.false_positive) if (self.true_positive + self.false_positive) > 0 else 0.0
        recall_failure = self.true_positive / (self.true_positive + self.false_negative) if (self.true_positive + self.false_negative) > 0 else 0.0
        
        # Precision and Recall for success detection
        precision_success = self.true_negative / (self.true_negative + self.false_negative) if (self.true_negative + self.false_negative) > 0 else 0.0
        recall_success = self.true_negative / (self.true_negative + self.false_positive) if (self.true_negative + self.false_positive) > 0 else 0.0
        
        return {
            'total_episodes': total,
            'accuracy': accuracy,
            'true_negative': self.true_negative,
            'false_positive': self.false_positive,
            'false_negative': self.false_negative, 
            'true_positive': self.true_positive,
            'precision_failure': precision_failure,
            'recall_failure': recall_failure,
            'precision_success': precision_success,
            'recall_success': recall_success,
            'episodes_with_failure_detection': self.episodes_with_failure_detection,
            'total_failure_detections': self.total_failure_detections,
            'avg_detections_per_episode': self.total_failure_detections / total if total > 0 else 0.0
        }


def print_confusion_matrix(metrics: Dict, log_file):
    """Print confusion matrix and metrics in a nice format"""
    output = f"""
=== FAILURE DETECTION METRICS ===
Total Episodes: {metrics['total_episodes']}
Overall Accuracy: {metrics['accuracy']:.4f}

Confusion Matrix:
                 Predicted
                Success  Failure
Actual Success    {metrics['true_negative']:4d}     {metrics['false_positive']:4d}   
Actual Failure    {metrics['false_negative']:4d}     {metrics['true_positive']:4d}   

Precision:
  Success: {metrics['precision_success']:.4f}
  Failure: {metrics['precision_failure']:.4f}

Recall:
  Success: {metrics['recall_success']:.4f}  
  Failure: {metrics['recall_failure']:.4f}

Failure Detection Stats:
  Episodes with failure detected: {metrics['episodes_with_failure_detection']}/{metrics['total_episodes']}
  Total failure detections: {metrics['total_failure_detections']}
  Avg detections per episode: {metrics['avg_detections_per_episode']:.2f}
===================================
"""
    print(output)
    log_file.write(output + "\n")
    log_file.flush()


@draccus.wrap()
def eval_failure_detection(cfg: FailureEvalConfig) -> None:
    assert cfg.model_checkpoint != "", "cfg.model_checkpoint must not be empty!"
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Load our failure detection model
    print(f"Loading failure detection model from {cfg.model_checkpoint}")
    model = OpenVLAWithFailureDetection.from_pretrained(cfg.model_checkpoint)
    model.eval()
    
    # Set action un-normalization key
    unnorm_key = cfg.task_suite_name
    if hasattr(model.openvla_model, 'norm_stats'):
        if unnorm_key not in model.openvla_model.norm_stats and f"{unnorm_key}_no_noops" in model.openvla_model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.openvla_model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    
    # Initialize logging
    run_id = f"FAILURE-EVAL-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")
    
    # Initialize metrics tracker
    metrics_tracker = FailureMetricsTracker()
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    
    # Get expected image dimensions
    resize_size = 224  # OpenVLA uses 224x224 images
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        
        # Get default LIBERO initial states  
        initial_states = task_suite.get_task_init_states(task_id)
        
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, "openvla", resolution=256, render_gui=False)
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            
            # Reset environment
            env.reset()
            
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            # Setup
            t = 0
            replay_images = []
            failure_detections = []
            failure_probs = []
            
            # Set max steps based on task suite
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                max_steps = 300
                
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                        t += 1
                        continue
                        
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)
                    
                    # Convert numpy array to PIL Image for our model
                    from PIL import Image
                    if isinstance(img, np.ndarray):
                        img_pil = Image.fromarray(img.astype(np.uint8))
                    else:
                        img_pil = img
                    
                    # Get action and failure prediction from our model
                    result = model.predict_action_and_failure(
                        image=img_pil,
                        task_description=task_description,
                        unnorm_key=unnorm_key,
                        center_crop=cfg.center_crop
                    )
                    
                    action = result['action']
                    failure_prob = result['failure_prob']
                    is_failure = result['is_failure']
                    
                    # Store failure detection info
                    failure_probs.append(failure_prob)
                    failure_detections.append(is_failure)
                    
                    # Log failure detection
                    if is_failure:
                        print(f"  Step {t}: FAILURE DETECTED (prob: {failure_prob:.4f})")
                        log_file.write(f"  Step {t}: FAILURE DETECTED (prob: {failure_prob:.4f})\n")
                    
                    # Normalize and align gripper action
                    action = normalize_gripper_action(action, binarize=True)
                    action = invert_gripper_action(action)
                    
                    # Execute action
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    
                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break
                    
            task_episodes += 1
            total_episodes += 1
            
            # Analyze failure detections for this episode
            episode_had_failure_detection = any(failure_detections)
            num_failure_detections = sum(failure_detections)
            avg_failure_prob = np.mean(failure_probs) if failure_probs else 0.0
            
            # Update metrics
            metrics_tracker.update(
                episode_success=done,
                failure_detected=episode_had_failure_detection,
                num_failure_detections=num_failure_detections
            )
            
            # Save replay video
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )
            
            # Log episode results with failure detection info
            failure_status = "FAILURE DETECTED" if episode_had_failure_detection else "NO FAILURE"
            print(f"Episode {total_episodes}:")
            print(f"  Success: {done}")
            print(f"  Failure Detection: {failure_status}")
            print(f"  Failure Detections: {num_failure_detections}/{len(failure_detections)} steps")
            print(f"  Avg Failure Prob: {avg_failure_prob:.4f}")
            
            log_file.write(f"Episode {total_episodes}:\n")
            log_file.write(f"  Success: {done}\n")
            log_file.write(f"  Failure Detection: {failure_status}\n") 
            log_file.write(f"  Failure Detections: {num_failure_detections}/{len(failure_detections)} steps\n")
            log_file.write(f"  Avg Failure Prob: {avg_failure_prob:.4f}\n")
            
            # Print running success rate
            success_rate = total_successes / total_episodes * 100
            print(f"Running success rate: {total_successes}/{total_episodes} ({success_rate:.1f}%)")
            log_file.write(f"Running success rate: {total_successes}/{total_episodes} ({success_rate:.1f}%)\n")
            log_file.flush()
            
        # Task completed
        task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
        print(f"Task '{task_description}' completed: {task_successes}/{task_episodes} ({task_success_rate*100:.1f}%)")
        log_file.write(f"Task '{task_description}' completed: {task_successes}/{task_episodes} ({task_success_rate*100:.1f}%)\n")
        
    # Final results
    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    print(f"\n=== FINAL RESULTS ===")
    print(f"Overall Success Rate: {total_successes}/{total_episodes} ({final_success_rate*100:.1f}%)")
    log_file.write(f"\n=== FINAL RESULTS ===\n")
    log_file.write(f"Overall Success Rate: {total_successes}/{total_episodes} ({final_success_rate*100:.1f}%)\n")
    
    # Print confusion matrix and failure detection metrics
    final_metrics = metrics_tracker.get_metrics()
    if final_metrics:
        print_confusion_matrix(final_metrics, log_file)
    
    log_file.close()
    print(f"Evaluation complete! Results saved to: {local_log_filepath}")


if __name__ == "__main__":
    eval_failure_detection()