from django.urls import path
from core import views

urlpatterns = [
  path('', views.home_page, name='home'),
  path('api/chat/', views.chat_with_llama, name='chat_with_llama'),
  
]