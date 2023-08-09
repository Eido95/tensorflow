```
$ pip install tensorflow
$ pip install pillow
$ pip install matplotlib
$ pip install flask

curl -X POST -H "Content-Type: image/png" --data-binary "@/path/to/digit.png" http://localhost:5000/predict/digit
```