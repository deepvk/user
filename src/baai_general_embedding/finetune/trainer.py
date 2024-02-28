from sentence_transformers import SentenceTransformer, models
from transformers.trainer import *
import wandb
import os


def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    word_embedding_model = models.Transformer(ckpt_dir)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    if normlized:
        normlize_layer = models.Normalize()
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], device='cpu')
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cpu')
    model.save(ckpt_dir)


class BiTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        if self.is_world_process_zero():
            save_ckpt_for_sentence_transformers(output_dir,
                                                pooling_mode=self.args.sentence_pooling_method,
                                                normlized=self.args.normlized)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
#         print(inputs)
        outputs = model(**inputs)
        loss = outputs.loss
    
        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval") -> Dict[str, float]:
        # memory metrics - must set up as early as possible
#         self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return

        args = self.args
        self.model.eval()
        
        results = {'mean': []}
        
        for name in self.eval_dataset.keys():
            dataset = self.eval_dataset[name]
            
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                pin_memory=True,
                collate_fn=self.data_collator,
            )

            losses = []

            for inputs in dataloader:
                inputs['query']['input_ids'] = inputs['query']['input_ids'].cuda()
                inputs['query']['attention_mask'] = inputs['query']['attention_mask'].cuda()
                inputs['passage']['input_ids'] = inputs['passage']['input_ids'].cuda()
                inputs['passage']['attention_mask'] = inputs['passage']['attention_mask'].cuda()


                losses.append(self.compute_loss(self.model, inputs))

            loss = float(torch.mean(torch.tensor(losses)))
#             print(name, mean_loss)
            

            wandb.log({f'val_loss_{os.path.basename(name)}': loss})
            results[os.path.basename(name)] = loss
            results['mean'].append(loss)
            
        results['mean'] = torch.mean(torch.tensor(results['mean']))
        wandb.log({f'val_loss_mean': results['mean']})
            
        return results
