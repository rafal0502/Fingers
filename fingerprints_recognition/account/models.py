from django.db import models

# Create your models here.
from django.conf import settings

class Profile(models.Model):
	user = models.OneToOneField(settings.AUTH_USER_MODEL)
	fingerprint_photo_1 = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)
	fingerprint_photo_2 = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)
	fingerprint_photo_3 = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)
	fingerprint_photo_4 = models.ImageField(upload_to='users/%Y/%m/%d', blank=True)

	def __str__(self):
		return 'Profil użytkownika {}.'.format(self.user.username)
