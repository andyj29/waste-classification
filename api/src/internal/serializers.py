from rest_framework import serializers
from .models import Image, Location, WasteCategory


class ImageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Image
        fields = ('__all__')


class WasteCategorySerializer(serializers.ModelSerializer):

    class Meta:
        model = WasteCategory
        fields = ('__all__')


class LocationSerializer(serializers.ModelSerializer):
    category = WasteCategorySerializer(many=True, read_only=True)

    class Meta:
        model = Location
        fields = ('__all__')
