# Generated by Django 3.1.3 on 2021-05-06 03:41

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('address', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.RenameField(
            model_name='profile',
            old_name='date_created',
            new_name='dateCreated',
        ),
        migrations.RenameField(
            model_name='raspdevice',
            old_name='date_created',
            new_name='dateAdded',
        ),
        migrations.RemoveField(
            model_name='car',
            name='driver',
        ),
        migrations.RemoveField(
            model_name='raspdevice',
            name='lastActive',
        ),
        migrations.AddField(
            model_name='profile',
            name='address',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='raspdevice',
            name='ipaddress',
            field=models.CharField(blank=True, choices=[('online', 'online'), ('offline', 'offline')], max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='car',
            name='company',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='accounts.company'),
        ),
        migrations.AddField(
            model_name='profile',
            name='company',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='accounts.company'),
        ),
    ]