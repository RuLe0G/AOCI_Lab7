using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace Lab7
{
    public partial class Form1 : Form
    {
        public Image<Bgr, byte> BaseImage;
        public Image<Bgr, byte> TwistedImage;

        public Form1()
        {
            InitializeComponent();
            imageBox3.Visible = false;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var openFileDialog = new OpenFileDialog();
            var result = openFileDialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                var fileName = openFileDialog.FileName;
                BaseImage = new Image<Bgr, byte>(fileName);

                imageBox1.Image = BaseImage.Resize(640, 480, Inter.Linear);
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            var openFileDialog = new OpenFileDialog();
            var result = openFileDialog.ShowDialog();
            if (result == DialogResult.OK)
            {
                var fileName = openFileDialog.FileName;
                TwistedImage = new Image<Bgr, byte>(fileName);

                imageBox2.Image = TwistedImage.Resize(640, 480, Inter.Linear);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var detector = new GFTTDetector(40, 0.01, 5, 3, true);
            var gfp1 = detector.Detect(BaseImage.Convert<Gray, byte>().Mat);


            var srcPoints = new PointF[gfp1.Length];
            for (var i = 0; i < gfp1.Length; i++)
                srcPoints[i] = gfp1[i].Point;

            CvInvoke.CalcOpticalFlowPyrLK(
                BaseImage.Convert<Gray, byte>().Mat,
                TwistedImage.Convert<Gray, byte>().Mat,
                srcPoints,
                new Size(20, 20),
                5,
                new MCvTermCriteria(20, 1),
                out var destPoints,
                out _,
                out _
            );


            var holographyMatrix = CvInvoke.FindHomography(destPoints, srcPoints, RobustEstimationAlgorithm.LMEDS);
            var destImage = new Image<Bgr, byte>(BaseImage.Size);
            CvInvoke.WarpPerspective(TwistedImage, destImage, holographyMatrix, destImage.Size);


            var output = BaseImage.Clone();

            foreach (var p in srcPoints) CvInvoke.Circle(output, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 5);
            imageBox1.Image = output.Resize(640, 480, Inter.Linear);

            var output2 = destImage.Clone();

            foreach (var p in destPoints) CvInvoke.Circle(output2, Point.Round(p), 3, new Bgr(Color.Blue).MCvScalar, 5);

            imageBox2.Image = output2.Resize(640, 480, Inter.Linear);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            var detector = new GFTTDetector(40, 0.01, 5, 3, true);

            var baseImgGray = BaseImage.Convert<Gray, byte>();
            var twistedImgGray = TwistedImage.Convert<Gray, byte>();

            var descriptor = new Brisk();

            var gfp1 = new VectorOfKeyPoint();
            var baseDesc = new UMat();
            var bimg = twistedImgGray.Mat.GetUMat(AccessType.Read);

            var gfp2 = new VectorOfKeyPoint();
            var twistedDesc = new UMat();
            var timg = baseImgGray.Mat.GetUMat(AccessType.Read);

            detector.DetectRaw(bimg, gfp1);
            descriptor.Compute(bimg, gfp1, baseDesc);

            detector.DetectRaw(timg, gfp2);
            descriptor.Compute(timg, gfp2, twistedDesc);

            var matcher = new BFMatcher(DistanceType.L2);

            var matches = new VectorOfVectorOfDMatch();
            matcher.Add(baseDesc);
            matcher.KnnMatch(twistedDesc, matches, 2);
            var mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(255));
            Features2DToolbox.VoteForUniqueness(matches, 0.8, mask);

            var res = new Image<Bgr, byte>(BaseImage.Size);
            Features2DToolbox.VoteForSizeAndOrientation(gfp1, gfp1, matches, mask, 1.5, 20);
            Features2DToolbox.DrawMatches(TwistedImage, gfp1, BaseImage, gfp2, matches, res, new MCvScalar(255, 0, 0),
                new MCvScalar(255, 0, 0), mask);

            imageBox3.Image = res.Resize(1280, 480, Inter.Linear);

            var holography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(gfp1, gfp2, matches, mask, 2);

            var destImage = new Image<Bgr, byte>(BaseImage.Size);
            CvInvoke.WarpPerspective(TwistedImage, destImage, holography, destImage.Size);

            imageBox2.Image = destImage.Resize(640, 480, Inter.Linear);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            imageBox3.Visible = checkBox1.Checked;
        }
    }
}