from django.db import models

# Create your models here.
class IMG(models.Model):
    name = models.CharField(max_length=50, default='usr')
    img = models.FileField(upload_to='img/')
    
    def __str__(self):
        return self.name
