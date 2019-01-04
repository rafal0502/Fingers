from django.contrib import admin
from .models import Profile
# Register your models here.
class ProfileAdmin(admin.ModelAdmin):
	list_display = ['user', 'fingerprint_photo_1', 'fingerprint_photo_2', 'fingerprint_photo_3', 'fingerprint_photo_4']


admin.site.register(Profile, ProfileAdmin)

