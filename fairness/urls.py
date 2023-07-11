from django.urls import path
from . import views

app_name = 'fairness'
urlpatterns = [
  path("", views.index),
  path("start/", views.start)
]