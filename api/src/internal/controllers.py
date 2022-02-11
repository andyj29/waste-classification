from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ParseError, NotFound
from rest_framework import status
from .serializers import LocationSerializer, WasteCategorySerializer
from .models import Image, Location, WasteCategory
from .services import classify_image, get_area_from_lat_long, is_in_gta
from .constants import Label, Area


class ClassifyImage(APIView):
    def post(self, request, *args, **kwargs):

        try:
            file = self.request.data['file']
        except KeyError:
            raise ParseError('Request has no image file')

        image = Image.objects.create(image=file)

        prediction = classify_image(image.get_url)
        x = prediction['label']
        print(x)

        category = WasteCategory.objects.filter(type=x).first()

        category_serializer = WasteCategorySerializer(category)

        try:
            latitude = self.request.data['lat']
            longitude = self.request.data['long']
        except KeyError:
            raise ParseError('Request has no longitude/latitude')
        
        location = get_area_from_lat_long(latitude, longitude)
        print(location)
 
        if is_in_gta(location):
            queryset = Location.objects.filter(area=location, category=category)
            loc_list = LocationSerializer(queryset, many=True)
        else:
            raise NotFound('Location is out of range')

        data = {
            'prediction': prediction,
            'category': category_serializer.data,
            'locations': loc_list.data
        }

        return Response(data, status=status.HTTP_202_ACCEPTED)
