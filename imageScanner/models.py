from django.db import models


class UploadImage(models.Model):
    user_complaint_prose = models.TextField(max_length=250, null=True, blank=True)
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.image
