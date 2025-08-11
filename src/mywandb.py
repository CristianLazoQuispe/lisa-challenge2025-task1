import os
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/va0831/Projects/lisa-challenge2025/.env")



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"  # Aquí se guardarán los archivos temporales y logs
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"




def init_wandb(args,experiment_name):

    PROJECT_WANDB = os.getenv("PROJECT_WANDB")
    ENTITY = os.getenv("ENTITY")
    print("Reconfiguration")
    if (args.tag_wandb != ""):
        if ("," in args.tag_wandb):
            tag_wandb = args.tag_wandb.split(",")
        elif ("-" in args.tag_wandb):
            tag_wandb = args.tag_wandb.split("-")

        else:
            tag_wandb = [args.tag_wandb]
    else:
        tag_wandb = ["2025_lisa"]
    # MARK: TRAINING PREPARATION AND MODULES
    run = wandb.init(project=PROJECT_WANDB, 
                    entity=ENTITY,
                    config=args,
                    name=experiment_name,
                    job_type="model-training",
                    allow_val_change=True,
                    save_code=True,
                    resume=None,
                    id= None,
                    tags= tag_wandb)
    return run
