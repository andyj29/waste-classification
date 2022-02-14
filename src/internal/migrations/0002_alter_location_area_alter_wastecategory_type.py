# Generated by Django 4.0.2 on 2022-02-14 00:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('internal', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='location',
            name='area',
            field=models.CharField(choices=[('Ajax', 'Ajax'), ('Aurora', 'Aurora'), ('Brampton', 'Brampton'), ('Brock', 'Brock'), ('Burlington', 'Burlington'), ('Caledon', 'Caledon'), ('Clarington', 'Clarington'), ('EastGwillimbury', 'East Gwillimbury'), ('Georgina', 'Georgina'), ('HaltonHills', 'Halton Hills'), ('King', 'King'), ('Markham', 'Markham'), ('Milton', 'Milton'), ('Mississauga', 'Mississauga'), ('Newmarket', 'Newmarket'), ('Oakville', 'Oakville'), ('Oshawa', 'Oshawa'), ('Pickering', 'Pickering'), ('RichmondHill', 'Richmond Hill'), ('Scugog', 'Scugog'), ('Toronto', 'Toronto'), ('Uxbridge', 'Uxbridge'), ('Vaughan', 'Vaughan'), ('Whitby', 'Whitby'), ('WhitchurchStouffville', 'Whitchurch Stouffville')], max_length=50),
        ),
        migrations.AlterField(
            model_name='wastecategory',
            name='type',
            field=models.CharField(choices=[('battery', 'battery'), ('biological', 'biological'), ('brown_glass', 'brown glass'), ('cardboard', 'cardboard'), ('clothes', 'clothes'), ('green_glass', 'green glass'), ('metal', 'metal'), ('paper', 'paper'), ('plastic', 'plastic'), ('shoes', 'shoes'), ('trash', 'trash'), ('white_glass', 'white glass')], max_length=50),
        ),
    ]