from django import forms
from .models import TranslationJob


class TranslationJobForm(forms.ModelForm):
    class Meta:
        model = TranslationJob
        fields = ['original_image', 'source_language', 'target_language']
