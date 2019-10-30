/*
 * 4dface: Real-time 3D face tracking and reconstruction from 2D video.
 *
 * File: apps/4dface.cpp
 *
 * Copyright 2015, 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "helpers.hpp"

#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/Mesh.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/fitting/contour_correspondence.hpp"
#include "eos/fitting/closest_edge_fitting.hpp"
#include "eos/fitting/RenderingParameters.hpp"

#include "rcr/model.hpp"
#include "cereal/cereal.hpp"

#include "glm/gtc/quaternion.hpp"

#include "Eigen/Dense"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"

#include <vector>
#include <iostream>

using namespace eos;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using cv::Rect;
using std::cout;
using std::endl;
using std::vector;
using std::string;

/**
 * This app demonstrates facial landmark tracking, estimation of the 3D pose
 * and fitting of the shape model of a 3D Morphable Model from a video stream,
 * and merging of the face texture.
 */
int main(int argc, char *argv[]) {
    fs::path modelfile, inputvideo, facedetector, landmarkdetector, mappingsfile, contourfile, edgetopologyfile, blendshapesfile;
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
                ("help,h",
                 "display the help message")
                ("morphablemodel,m",
                 po::value<fs::path>(&modelfile)->required()->default_value("../share/sfm_shape_3448.bin"),
                 "a Morphable Model stored as cereal BinaryArchive")
                ("facedetector,f", po::value<fs::path>(&facedetector)->required()->default_value(
                        "../share/haarcascade_frontalface_alt2.xml"),
                 "full path to OpenCV's face detector (haarcascade_frontalface_alt2.xml)")
                ("landmarkdetector,l", po::value<fs::path>(&landmarkdetector)->required()->default_value(
                        "../share/face_landmarks_model_rcr_68.bin"),
                 "learned landmark detection model")
                ("mapping,p",
                 po::value<fs::path>(&mappingsfile)->required()->default_value("../share/ibug_to_sfm.txt"),
                 "landmark identifier to model vertex number mapping")
                ("model-contour,c", po::value<fs::path>(&contourfile)->required()->default_value(
                        "../share/sfm_model_contours.json"),
                 "file with model contour indices")
                ("edge-topology,e", po::value<fs::path>(&edgetopologyfile)->required()->default_value(
                        "../share/sfm_3448_edge_topology.json"),
                 "file with model's precomputed edge topology")
                ("blendshapes,b", po::value<fs::path>(&blendshapesfile)->required()->default_value(
                        "../share/expression_blendshapes_3448.bin"),
                 "file with blendshapes")
                ("input,i", po::value<fs::path>(&inputvideo),
                 "input video file. If not specified, camera 0 will be used.");
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            cout << "Usage: 4dface [options]" << endl;
            cout << desc;
            cout << desc;
            return EXIT_FAILURE;
        }
        po::notify(vm);
        po::notify(vm);
    }
    catch (const po::error &e) {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        return EXIT_FAILURE;
    }

    // Load the Morphable Model and the LandmarkMapper:
    const morphablemodel::MorphableModel morphable_model = morphablemodel::load_model(modelfile.string());
    const core::LandmarkMapper landmark_mapper = mappingsfile.empty() ? core::LandmarkMapper() : core::LandmarkMapper(
            mappingsfile.string());

    const fitting::ModelContour model_contour = contourfile.empty() ? fitting::ModelContour()
                                                                    : fitting::ModelContour::load(contourfile.string());
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile.string());

    rcr::detection_model rcr_model;
    // Load the landmark detection model:
    try {
        rcr_model = rcr::load_detection_model(landmarkdetector.string());
    }
    catch (const cereal::Exception &e) {
        cout << "Error reading the RCR model " << landmarkdetector << ": " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Load the face detector from OpenCV:
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(facedetector.string())) {
        cout << "Error loading the face detector " << facedetector << "." << endl;
        return EXIT_FAILURE;
    }

    cv::VideoCapture cap;
    if (inputvideo.empty()) {
        cap.open(0); // no file given, open the default camera
    } else {
        cap.open(inputvideo.string());
    }
    if (!cap.isOpened()) {
        cout << "Couldn't open the given file or camera 0." << endl;
        return EXIT_FAILURE;
    }

    const morphablemodel::Blendshapes blendshapes = morphablemodel::load_blendshapes(blendshapesfile.string());

    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile.string());

    Mat frame, unmodified_frame;

    bool have_face = false;
    rcr::LandmarkCollection<Vec2f> current_landmarks;
    WeightedIsomapAveraging isomap_averaging(30.f); // merge all triangles that are facing <60� towards the camera
    PcaCoefficientMerging pca_shape_merging;

    for (;;) {
        cap >> frame; // get a new frame from camera
        if (frame.empty()) { // stop if we're at the end of the video
            break;
        }

        // We do a quick check if the current face's width is <= 50 pixel. If it is, we re-initialise the tracking with the face detector.
        if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
            // Reinitialising because the face bounding-box width is <= 50 px
            have_face = false;
        }

        unmodified_frame = frame.clone();

        if (!have_face) {
            // Run the face detector and obtain the initial estimate using the mean landmarks:
            vector<Rect> detected_faces;
            face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
            if (detected_faces.empty()) {
                cout << 0.f << endl;
                usleep(30000);
                continue;
            }
            cv::rectangle(frame, detected_faces[0], {255, 0, 0});
            // Rescale the V&J facebox to make it more like an ibug-facebox:
            // (also make sure the bounding box is square, V&J's is square)
            Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

            current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);

            have_face = true;
        } else {
            // We already have a face - track and initialise using the enclosing bounding
            // box from the landmarks from the last frame:
            auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
            enclosing_bbox = make_bbox_square(enclosing_bbox);
            current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
        }

        // Fit the 3DMM:
        fitting::RenderingParameters rendering_params;
        vector<float> shape_coefficients, blendshape_coefficients;
        vector<Eigen::Vector2f> image_points;
        core::Mesh mesh;
        std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(morphable_model, blendshapes,
                                                                       rcr_to_eos_landmark_collection(
                                                                               current_landmarks), landmark_mapper,
                                                                       unmodified_frame.cols, unmodified_frame.rows,
                                                                       edge_topology, ibug_contour, model_contour, 3, 5,
                                                                       15.0f, cpp17::nullopt, shape_coefficients,
                                                                       blendshape_coefficients, image_points);

        cout << glm::eulerAngles(rendering_params.get_rotation())[1] << endl;
        usleep(30000);
    }

    return EXIT_SUCCESS;
}
