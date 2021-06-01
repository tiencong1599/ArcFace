# Face Recognition with InsightFace
Recognize and manipulate faces with Python and its support libraries.  
The project uses [MTCNN](https://github.com/ipazc/mtcnn) for detecting faces, then applies a simple alignment for each detected face and feeds those aligned faces into embeddings model provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, a softmax classifier was put on top of embedded vectors for classification task.

## Getting started
### Requirements
- Python 3.6
- Virtualenv
- python-pip
- mx-net
- tensorflow
- macOS, Linux or Windows 10
### Installing in Linux or MacOs
Check update:
```
sudo apt-get update
```
Install python:
```
sudo apt-get install python3.6
```
Install pip:
```
sudo apt install python3-pip
```
Most of the necessary libraries were installed and stored in `env/` folder, so what we need is installing `virtualenv` to use this enviroment.  
Install virtualenv:
```
sudo pip3 install virtualenv virtualenvwrapper
```
### Installing in Windows 10
Install python 3.6 at [here](https://www.python.org/downloads/release/python-360/).
Install virtualenv:
```
pip install virtualenv
```
After that go to env folder and activate virtualenv:
```angular2html
cd /path/to/virtualenv/
```
Activate virtualenv:
```angular2html
call virtualenv/Scripts/activate.bat
```
Install requirements.txt:
```angular2html
pip install -r requirements.txt
```
Go to src folder:
```angular2html
cd src
```
Run main.py to launch GUI.
```angular2html
python main.py
```
## Usage
After launch GUI of ArcFace, we have this interface.

