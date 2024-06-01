# UnblurNet
Removes different types and intensities of blurs from images

## Setup a virtual environment
- Install python3-virtualenv pakage system-wide
  `sudo apt-get install python3-virtualenv`
- Create a virtual environment inside the project directory
  `python3 -m venv .venv`
- Activate your venv
  `source .venv/bin/activate`

## Install project dependencies
  `pip install -r requirements.txt`

## Get the [dataset](https://www.kaggle.com/kwentar/blur-dataset) from kaggle
  After downloading the dataset setup the project directory in such a way
  ```
  UnblurNet
  ├───inputs
  │   ├───defocused_blurred
  │   ├───gaussian_blurred
  │   ├───motion_blurred
  │   └───sharp
  ├───outputs
  │   └───saved_images
  └───src
  ```

## Testing the model
- Currently the train.py script doesn't use the blurred images provided in the dataset.
- You can create a new directory with gaussian blurring using the utils/add_gaussian_blur.py script
  `python3 utils/add_gaussian_blur.py`
- Now we can run the src/train.py script
  `python3 src/train.py --epochs 100`
 
