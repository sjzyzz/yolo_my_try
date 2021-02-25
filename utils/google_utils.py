import requests
import torch
import os
from pathlib import Path


def attemp_download(weights):
    weights = str(weights).strip().replace("'", "")
    file = Path(weights).name.lower()

    msg = (
        weights
        + " missing, try downloading from https://github.com/ultralytics/yolov3/releases/"
    )
    response = requests.get(
        "https://api.github.com/repos/ultralytics/yolov3/releases/latest"
    ).json()
    assets = [x["name"] for x in response["assets"]]
    redundant = False

    if file in assets and not os.path.isfile(weights):
        try:
            tag = response["tag_name"]
            url = (
                f"https://github.com/ultralytics/yolov3/releases/download/{tag}/{file}"
            )
            print("Downloading %s to %s..." % (url, weights))
            torch.hub.download_url_to_file(url, weights)
            assert os.path.exist(weights) and os.path.getsize(weights) > 1e6
        except Exception as e:
            print("Downloading error: %s" % e)
            assert redundant, "No secondary mirror"
            url = "https://storage.googleapis.com/ultralytics/yolov3/ckpt/" + file
            print("Downloading %s to %s..." % (url, weights))
            r = os.system("curl -L %s -o %s" % (url, weights))
        finally:
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1e6):
                os.remove(weights) if os.path.exists(weights) else None
                print("ERROR: Downloading failure: %s" % msg)
            print("")
            return
