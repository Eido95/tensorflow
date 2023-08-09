## MNIST

### Model Setup

```
$ pip install tensorflow
$ pip install pillow
$ pip install matplotlib
```

### App Setup

```
$ pip install flask
```

### App Usage

```
curl -X POST -H "Content-Type: image/png" --data-binary "@/path/to/digit.png" http://localhost:5000/predict/digit
```

