package facedetector;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_core.cvClearMemStorage;
import static org.bytedeco.javacpp.opencv_core.cvCopy;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvPoint;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.cvRectangle;
import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;

import org.bytedeco.javacpp.opencv_core.CvMemStorage;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvScalar;
import org.bytedeco.javacpp.opencv_core.CvSeq;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

/**
 * Face detector system that get an image file and uses haarcascade algorithm to
 * find a face in it.
 * 
 * @author giuliana.bezerra
 *
 */
public class FaceDetection {

	public static final String XML_FILE = "src/main/resources/haarcascade_frontalface_alt.xml";
	public static final String IMG_PATH = "src/main/resources/lena.png";

	public static void main(String[] args) {
		IplImage img = cvLoadImage(IMG_PATH);
		detect(img);
	}

	public static void detect(IplImage src) {
		CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(XML_FILE));
		CvMemStorage storage = CvMemStorage.create();
		CvSeq faces = cvHaarDetectObjects(src, cascade, storage, 1.5, 3, CV_HAAR_DO_CANNY_PRUNING);
		cvClearMemStorage(storage);
		int totalFaces = faces.total();
		System.out.print(totalFaces); // How many faces were found? It will be
										// considered the first one.
		drawRectangule(src, faces);
		cropFaceImage(src, faces);
		showImage(src);
	}

	private static void cropFaceImage(IplImage src, CvSeq faces) {
		CvRect rectangule = new CvRect(cvGetSeqElem(faces, 0));
		cvSetImageROI(src, rectangule);
		IplImage faceImage = cvCreateImage(cvGetSize(src), src.depth(), src.nChannels());
		cvCopy(src, faceImage);
		cvSaveImage("face.png", faceImage);
	}

	private static void drawRectangule(IplImage src, CvSeq faces) {
		CvRect rectangule = new CvRect(cvGetSeqElem(faces, 0));
		cvRectangle(src, cvPoint(rectangule.x(), rectangule.y()),
				cvPoint(rectangule.width() + rectangule.x(), rectangule.height() + rectangule.y()), CvScalar.RED, 2,
				CV_AA, 0);
		rectangule.close();
	}

	private static void showImage(IplImage src) {
		OpenCVFrameConverter.ToIplImage converter = new OpenCVFrameConverter.ToIplImage();
		CanvasFrame frameGeral = new CanvasFrame("Face", CanvasFrame.getDefaultGamma());
		Frame frameConverted = converter.convert(src);
		frameGeral.showImage(frameConverted);
	}
}