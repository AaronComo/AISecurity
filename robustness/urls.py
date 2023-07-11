from django.urls import path
from . import views

app_name = 'robustness'
urlpatterns = [
  # path('', views.index),
  path("", views.uploadImg, name="upload"),
  path("test/", views.testRobustness, name="test"),
  path("token/", views.token, name="token"),
]