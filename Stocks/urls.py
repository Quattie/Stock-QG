from django.urls import path
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('', views.home, name='stock-home'),
    path('about/', views.about, name='stock-about'),
    path('recurrent/', views.recurrent, name='stock-recurrent'),
    path('random-forests/', views.randomForest, name='stock-random-forests'),
    path('crypto/', views.crypto, name='stock-crypto'),
    path('crypto-model/', views.cryptoModel, name='stock-crypto-model')
]

urlpatterns += staticfiles_urlpatterns()
