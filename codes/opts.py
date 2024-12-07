import argparse

parser = argparse.ArgumentParser()
parser.add_argument( "--expeiment_name",help="name of the experiment",type=str,)
parser.add_argument("--config", help="name of the dataset", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-B/32")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=64, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=1024, type=int)
parser.add_argument("--evaluate_only",help="directly evaluate on the" "dataset without any training",action="store_true",)
parser.add_argument("--context_length",help="sets the context length of the clip model",default=32,type=int,)
parser.add_argument("--attr_dropout",help="add dropout to attributes",type=float,default=0.0,)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n",default=1,type=int,help="saves the model every n epochs; ""this is useful for validation/grid search",)
parser.add_argument("--save_model",help="indicate if you want to save the model state dict()",action="store_true")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps",help="number of gradient accumulation steps",default=1,type=int)

parser.add_argument("--logpath",default='/mnt/fast/nobackup/users/rl01003/ex_ideas/com_vlm/log/Discrete_condition_v2/tsm18/10/',type=str)


