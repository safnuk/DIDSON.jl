from modelmanager import run

from didson.model import DidsonModel

models = {
    "DIDSON": DidsonModel,
}

if __name__ == "__main__":
    run(models)
