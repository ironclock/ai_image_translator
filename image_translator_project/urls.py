"""
URL configuration for image_translator_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from translator.views import TranslationJobView, TranslationResultView
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', TranslationJobView.as_view(), name='upload'),
#     path('result/<int:job_id>/', TranslationResultView.as_view(),
#          name='translation_result'),
# ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', TranslationJobView.as_view(), name='upload'),
    path('result/<int:job_id>/', TranslationResultView.as_view(),
         name='translation_result'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
