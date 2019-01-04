# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('account', '0002_auto_20181212_0932'),
    ]

    operations = [
        migrations.RenameField(
            model_name='profile',
            old_name='fingerprint_photo',
            new_name='fingerprint_photo_1',
        ),
        migrations.AddField(
            model_name='profile',
            name='fingerprint_photo_2',
            field=models.ImageField(blank=True, upload_to='users/%Y/%m/%d'),
        ),
        migrations.AddField(
            model_name='profile',
            name='fingerprint_photo_3',
            field=models.ImageField(blank=True, upload_to='users/%Y/%m/%d'),
        ),
        migrations.AddField(
            model_name='profile',
            name='fingerprint_photo_4',
            field=models.ImageField(blank=True, upload_to='users/%Y/%m/%d'),
        ),
    ]
