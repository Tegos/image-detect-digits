from PIL import Image
from PIL import ImageChops
import datetime


def matchTemplate(searchImage, templateImage):
    minScore = -1000
    matching_xs = 0
    matching_ys = 0
    # convert images to "L" to reduce computation by factor 3 "RGB"->"L"
    searchImage = searchImage.convert(mode="L")
    templateImage = templateImage.convert(mode="L")
    searchWidth, searchHeight = searchImage.size
    templateWidth, templateHeight = templateImage.size
    # make a copy of templateImage and fill with color=1
    templateMask = Image.new(mode="L", size=templateImage.size, color=1)
    # loop over each pixel in the search image
    for xs in range(searchWidth - templateWidth + 1):
        for ys in range(searchHeight - templateHeight + 1):
            # for ys in range(10):
            # set some kind of score variable to "All equal"
            score = templateWidth * templateHeight
            # crop the part from searchImage
            searchCrop = searchImage.crop((xs, ys, xs + templateWidth, ys + templateHeight))
            diff = ImageChops.difference(templateImage, searchCrop)
            notequal = ImageChops.darker(diff, templateMask)
            countnotequal = sum(notequal.getdata())
            score -= countnotequal
            if minScore < score:
                minScore = score
                matching_xs = xs
                matching_ys = ys

    print "Location=", (matching_xs, matching_ys), "Score=", minScore
    im1 = Image.new('RGB', (searchWidth, searchHeight), (255, 255, 255))
    im1.paste(templateImage, ((matching_xs), (matching_ys)))
    # searchImage.show()
    # im1.show()
    im1.save('template_matched_in_search.png')


searchImage = Image.open("../res/555.png")
templateImage = Image.open("../res/search.png")
# searchImage = Image.open("search-500.png")
# templateImage = Image.open("template-80.png")
t1 = datetime.datetime.now()
matchTemplate(searchImage, templateImage)
delta = datetime.datetime.now() - t1
print"Time=%d.%d" % (delta.seconds, delta.microseconds)
print "end"
