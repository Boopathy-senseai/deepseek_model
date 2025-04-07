from django.urls import path
from .views import generate_text

urlpatterns = [
    path("generate/", generate_text, name="generate_text"),
]
