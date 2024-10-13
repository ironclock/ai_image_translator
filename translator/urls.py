from django.urls import path
from .views import TranslationJobView, TranslationResultView

urlpatterns = [
    path('', TranslationJobView.as_view(), name='upload'),
    path('result/<int:job_id>/', TranslationResultView.as_view(),
         name='translation_result'),
]
