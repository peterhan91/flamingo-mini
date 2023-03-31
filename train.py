"""
Use Huggingface Trainer with FlamingoModel.

This is a working demo script which you can adapt to your needs.
"""
import os
os.environ["WANDB_DISABLED"] = "true"
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random

import torch
# torch.backends.cuda.matmul.allow_tf32 = True
from torch.utils.data import Dataset
from torchvision import transforms as T
from cxr_dataset import CXRDataset

import transformers
from transformers import HfArgumentParser, CLIPImageProcessor
from transformers.trainer import Trainer, TrainingArguments
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor
from eval import evaluate_image_captioning  # don't ask me why this import works


logger = logging.getLogger(__name__)


# get images and annotations from https://cocodataset.org/#download
cxr_filepath = '/media/tianyu.han/mri-scratch/DeepLearning/Stanford_MIT_CHEST/MIMIC-CXR-v2.0/mimic-cxr/'
txt_filepath = './data/mimic_impressions.csv'
txt_filepath_val = './data/mimic_impressions_val.csv'


class CLIPImageTransform:
    """ experimental. A transform that does apply the transforms of a default CLIPFeatureExtractor """
    vision_processor: CLIPImageProcessor

    def __init__(self, clip_model_type: str):
        self.vision_processor = CLIPImageProcessor.from_pretrained(clip_model_type) # type: ignore

    def __call__(self, image) -> torch.Tensor:
        return self.vision_processor(images=image, 
                                     return_tensors="pt", padding=True)['pixel_values']

        
def prepare_training_dataset(config: FlamingoConfig):
    """ prepare a CocoCaptions training dataset """
    transform = T.Compose([
        T.Resize(config.img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),                       # add your favorite transforms
        T.ToTensor(),
        CLIPImageTransform(config.clip_model_type)
    ])

    def target_transform(captions):
        return f"{random.choice(['', ' '])}<image>{captions}<EOC></s>"

    return CXRDataset(img_path=cxr_filepath,
                    txt_path=txt_filepath, 
                    column='findings', 
                    transform=transform,
                    target_transform=target_transform)
    

def prepare_evaluation_dataset(config: FlamingoConfig):
    return CXRDataset(img_path=cxr_filepath,
                    txt_path=txt_filepath_val, 
                    column='findings', 
                    transform=T.Compose([T.Resize(config.img_size, interpolation=T.InterpolationMode.BICUBIC),
                                         T.ToTensor(),
                                        CLIPImageTransform(config.clip_model_type)])
                    )


class DataCollator:
    def __init__(self, config: FlamingoConfig):
        self.processor = FlamingoProcessor(config)
        
    def __call__(self, batch):
        pixel_values, sentences = zip(*batch)
        inputs = self.processor(text=sentences)
        pixel_values = torch.stack(pixel_values)
        
        return dict(
            pixel_values=pixel_values,
            labels=inputs['input_ids'],
            **inputs
        )


@dataclass
class FlamingoTrainingArguments(TrainingArguments):
    """ custom arguments """
    eval_coco_captioning_prefix: str = field(default="<image>A picture of")         # It's a common thing to do for COCO image captioning
    eval_coco_captioning_start: int = field(default=0)
    eval_coco_captioning_end: int = field(default=1000)
    

class FlamingoTrainer(Trainer):

    args: FlamingoTrainingArguments
    model: FlamingoModel
    processor: FlamingoProcessor
    eval_dataset: CXRDataset
    
    def evaluate(self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        pass
    
    
if __name__ == '__main__':
    parser = HfArgumentParser(FlamingoTrainingArguments)
    training_args: FlamingoTrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s', 
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(f'{args.output_dir}/out.log')
        ]    
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    logger.info(str(training_args))

    logger.info('loading model...')
    config = FlamingoConfig(
        clip_model_type='openai/clip-vit-base-patch32',
        lm='gpt2-xl',
        dim=1600,
        dim_visual=768,
        xattn_act='sqrelu',
        resampler_act='sqrelu'
    )
    model = FlamingoModel(config)
    model.train()

    #################################################################
    # datasets
    #################################################################
    logger.info('loading datasets...')
    train_dataset = prepare_training_dataset(config)
    eval_dataset = prepare_evaluation_dataset(config)
    
    #################################################################
    # optimizer, scheduler, trainer
    #################################################################
    # optimizer = AdamW(model.parameters_trainable(), training_args.learning_rate)
    # scheduler = get_constant_schedule_with_warmup(optimizer, training_args.warmup_steps)

    trainer = FlamingoTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(config),
        # optimizers=(optimizer, scheduler)
    )

    #################################################################
    # training loop
    #################################################################
    logger.info('start training.')

    if training_args.resume_from_checkpoint is not None:
        trainer.train(training_args.resume_from_checkpoint)
    else:
        trainer.train()