"""
Urls.py includes the url configurations of the application.
"""

from django.urls import path

from CNN.views import Predict,IndexView


app_name = 'CNN'

urlpatterns = [
    path("", IndexView.as_view(), name="index"),
    path('predict/', Predict.as_view(), name='predict'),
]