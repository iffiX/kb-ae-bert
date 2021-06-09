from ..utils.config import Config, QATrainConfig
from ..model.kb_ae import KBMaskedLMEncoder
from ..trainer.qa_trainer import QATrainer
from transformers import BatchEncoding



def train():
    # execute pipeline
    
    kb_encoder = KBMaskedLMEncoder(relation_size=3)
    config = QATrainConfig(train_dataset_path="squad", kb_encoder_trainable=False)

    trainer =  QATrainer(kb_encoder, config)

    train_loader = trainer.train_dataloader()


    for i, batch_data in enumerate(train_loader):
        
        batch = BatchEncoding(batch_data)
        loss = trainer.training_step(batch, i)

        print(loss)


# config = Config(pipeline=["qa", "kb_encoder"])

train()
