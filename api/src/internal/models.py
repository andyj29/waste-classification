from django.db import models
from .constants import Label, Area


class Image(models.Model):
    image = models.ImageField(upload_to='images')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class WasteCategory(models.Model):
    type = models.CharField(max_length=50, choices=Label.choices())
    desc = models.CharField(max_length=1024, blank=True, nullnull=True)
    recyclable = models.BooleanField(default=False)
    area = models.CharField(max_length=50, choices=Area.choices())


class Target(models.Model):
    address = models.CharField(max_length=255)
    instructions = models.TextField(blank=True, null=True)
    category = models.ForeignKey(WasteCategory, on_delete=models.SET_NULL, null=True)



