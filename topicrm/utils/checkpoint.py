from pathlib import Path
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import json

class CheckpointManager:
    FP32 = 'savedfp32'
    def __init__(self, path, checkpoints=None) -> None:
        self.path = Path(path)
        self.checkpoints = []
        if checkpoints is not None:
            for step, path in checkpoints:
                self.checkpoints.append((step, Path(path)))

    def save_checkpoint(self, step, model, accelerator=None):
        path = self.path / ('checkpoint-%d' % step)
        self.checkpoints.append((step, str(path)))
        hf_path = path/CheckpointManager.FP32
        if accelerator is not None:
            accelerator.wait_for_everyone()
            accelerator.save_state(path)
            if accelerator.is_main_process:
                hf_path.mkdir(parents=True, exist_ok=True)
                model.module.config.save_pretrained(hf_path)
                convert_zero_checkpoint_to_fp32_state_dict(str(path), hf_path/'pytorch_model.bin')
            accelerator.wait_for_everyone()
        else:
            hf_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(hf_path)
            #TODO save state of optimizer
        return hf_path
    
    def save_json(self):
        json.dump(self.checkpoints, open(self.path/'checkpoints.json', 'w'))
    
    @staticmethod
    def load_json(path):
        checkpoints = json.load(open(path, 'r'))
        return CheckpointManager(Path(path).parent, checkpoints=checkpoints)
    
    def get_model_checkpoints(self):
        for step, path in self.checkpoints:
            yield step, path/self.FP32