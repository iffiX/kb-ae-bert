from .base import QADataset, StaticMapDataset


class CoqaDataset(QADataset):

    def preprocess(self, split="train"):

        contexts = []
        questions = []
        answers = []
        indexes = []
        
        for idx, item in enumerate(self.dataset[split]):
            """
            {
                "answers": "{\"answer_end\": [179, 494, 511, 545, 879, 1127, 1128, 94, 150, 412, 1009, 1046, 643, -1, 764, 724, 125, 1384, 881, 910], \"answer_...",
                "questions": "[\"When was the Vat formally opened?\", \"what is the library for?\", \"for what subjects?\", \"and?\", \"what was started in 2014?\", \"ho...",
                "source": "wikipedia",
                "story": "\"The Vatican Apostolic Library (), more commonly called the Vatican Library or simply the Vat, is the library of the Holy See, l..."
            }
            """
            num_questions = len(item["questions"])

            for question_idx in range(num_questions):
                
                indexes.append(idx)
                contexts.append(item["story"])
                questions.append(item["question"][question_idx])

                answers.append({
                    "answer_start":item["answers"]["answer_start"][question_idx],
                    "answer_end":item["answers"]["answer_end"][question_idx],
                    "text":item["answers"]["input_text"][question_idx]
                    })