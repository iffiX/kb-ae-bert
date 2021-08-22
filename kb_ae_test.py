import os
from kb_ae_bert.utils.config import load_config
from kb_ae_bert.trainer.train import safe_load
from kb_ae_bert.trainer.kb_trainer import KBEncoderTrainer

if __name__ == "__main__":
    config = load_config("configs/train_kb_encoder_entity+relation.json")
    checkpoint_path = os.path.join(config.working_directory, "0", "checkpoint")
    stage_trainer, epoch = safe_load(KBEncoderTrainer, 0, config)
    model = stage_trainer.kb_model
