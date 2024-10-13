from django.db import models


class TranslationJob(models.Model):
    original_image = models.ImageField(upload_to='original_images/')
    translated_image = models.ImageField(
        upload_to='translated_images/', blank=True, null=True)
    source_language = models.CharField(max_length=50)
    target_language = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Translation job {self.id}: {self.source_language} to {self.target_language}"
