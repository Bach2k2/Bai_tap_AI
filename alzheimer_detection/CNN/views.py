"""
views.py includes the main business logic of the application.
Its role is to manage file upload, deletion and emotion predictions.
"""

import os
from os import listdir
from os.path import join
from os.path import isfile
from django.shortcuts import render,redirect
import requests

import keras
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.views.generic import TemplateView
from rest_framework import views
from rest_framework import status
from rest_framework.generics import get_object_or_404


from rest_framework.renderers import TemplateHTMLRenderer
from django.core.files.storage import FileSystemStorage 
from django.http import JsonResponse


class IndexView(TemplateView):
    """
    This is the index view of the website.
    :param template_name; Specifies the static display template file.
    """

    template_name = "index.html"


class Predict(views.APIView):
    """
    This class is used to making predictions.
    """
    template_name = "index.html"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_name = "cnn_model.keras"
        self.loaded_model = keras.models.load_model(
            os.path.join(settings.MODEL_ROOT, model_name)
        )
        self.class_dist = {
            "Non_Demented": 3200,
            "Mild_Demented": 896,
            "Moderate_Demented": 64,
            "Very_Mild_Demented": 2240,
        }

    def file_predict(self, filepath):
        """
        This function is used to elaborate the file used for the predictions.
        """
        image = tf.io.read_file(filepath)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, [128, 128])
        image = tf.cast(image, tf.float32) / 255.0

        pred = self.loaded_model.predict(tf.expand_dims(image, 0))[0]
        probs = tf.nn.softmax(pred).numpy()
        probs_dict = dict(zip(self.class_dist.keys(), map(float, probs)))  # Convert float32 to Python float
        return probs_dict


    
    def post(self, request):
        """
        This method is used to make predictions on uploaded files.
        """
        if 'file' not in request.FILES:
                return JsonResponse({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        filepath = fs.path(filename)
        print(filename)
        filepath = os.path.join(settings.MEDIA_ROOT, filename)
        predictions = self.file_predict(filepath)
        fs.delete(filename)  # Clean up the uploaded file after prediction
        return render(request, 'predict.html',{"predictions":predictions})
        # return response
