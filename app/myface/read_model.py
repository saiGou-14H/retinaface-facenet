import tensorflow.keras.models as model
from keras import Model, Input

if __name__ == '__main__':
    model_data=model.load_model("model_data/silent.h5",compile=False)
    print(model_data)
    input=Input(shape=(None, None, 3))
    model = Model(inputs=input, outputs=input)
    output=model_data.predict("image/2_qq2_001.jpg")
    print(output)