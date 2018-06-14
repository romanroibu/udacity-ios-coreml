//
//  RecognizerViewController.swift
//  SmartGroceryList
//
//  Created by Meghan Kane on 7/28/17.
//  Copyright © 2017 Meghan Kane. All rights reserved.
//

import UIKit
import Vision

extension VNImageRequestHandler {

    convenience init?(image: UIImage, options: [VNImageOption: Any] = [:]) {

        guard let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue)) else { return nil }

        if let cgImage = image.cgImage {
            self.init(cgImage: cgImage, orientation: orientation, options: options)
            return
        }

        if let ciImage = image.ciImage {
            self.init(ciImage: ciImage, orientation: orientation, options: options)
            return
        }

        return nil
    }
}

// MARK: - RecognizerViewController: UIViewController

class RecognizerViewController: UIViewController {

    // MARK: Outlets

    @IBOutlet var imageView: UIImageView!
    @IBOutlet var cameraButton: UIButton!
    @IBOutlet var predictionView: PredictionView!
    @IBOutlet var photoSourceView: UIView!
    @IBOutlet var tableView: UITableView!

    // MARK: Properties

    let CellReuseIdentifer = "GroceryItemCell"
    let imagePickerController = UIImagePickerController()
    var groceryItems: [String] = []
    var currentPrediction: String?

    // MARK: Life Cycle

    override func viewDidLoad() {
        super.viewDidLoad()
        imagePickerController.delegate = self
        predictionView.isHidden = true
        tableView.reloadData()
    }

    // MARK: Actions

    @IBAction func takePhoto() {
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            imagePickerController.sourceType = .camera
            present(imagePickerController, animated: true, completion: nil)
        } else {
            showCameraNotAvailableAlert()
        }
    }

    @IBAction func selectPhoto() {
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true, completion: nil)
    }

    @IBAction func addToList() {
        if let predictionToAdd = currentPrediction {
            groceryItems.append(predictionToAdd)
            tableView.reloadData()
            clearPrediction()
        }
    }

    @IBAction func rejectPrediction() {
        clearPrediction()
    }

    // MARK: Private

    private func setupPrediction(prediction: String) {
        predictionView.predictionResultLabel.text = prediction
        predictionView.isHidden = false
        photoSourceView.isHidden = true

        currentPrediction = prediction
    }

    private func clearPrediction() {
        predictionView.isHidden = true
        photoSourceView.isHidden = false
        predictionView.predictionResultLabel.text = nil
        imageView.image = nil
        currentPrediction = nil
    }

    private let classifier = Food101Net()

    private func classifyFood(image: UIImage) {

        let model = try! VNCoreMLModel(for: self.classifier.model)
        let request = VNCoreMLRequest(model: model, completionHandler: self.handleFoodClassificationResult)

        let handler = VNImageRequestHandler(image: image)!

        DispatchQueue.global(qos: .userInitiated).async {
            try! handler.perform([request])
        }
    }

    private func handleFoodClassificationResult(for request: VNRequest, error: Error?) {

        DispatchQueue.main.async {
            if let error = error {
                print(error)
                self.showRecognitionFailureAlert()
                return
            }

            guard let result = request.results?.lazy.flatMap({ $0 as? VNClassificationObservation }).first else {
                fatalError("No classification observation")
            }

            self.setupPrediction(prediction: result.identifier)
        }
    }

    private func showRecognitionFailureAlert() {
        let alertController = UIAlertController.init(title: "Recognition Failure", message: "Please try another image.", preferredStyle: .alert)
        alertController.addAction(UIAlertAction.init(title: "OK", style: .default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }

    private func showCameraNotAvailableAlert() {
        let alertController = UIAlertController.init(title: "Camera Not Available", message: nil, preferredStyle: .alert)
        alertController.addAction(UIAlertAction.init(title: "OK", style: .default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
}

// MARK: - RecognizerViewController: UITableViewDataSource

extension RecognizerViewController: UITableViewDataSource {

    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return groceryItems.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let groceryItemCell = tableView.dequeueReusableCell(withIdentifier: CellReuseIdentifer) as! GroceryItemTableViewCell
        if indexPath.row < groceryItems.count {
            let item = groceryItems[indexPath.row]
            groceryItemCell.nameLabel.text = item
        }

        return groceryItemCell
    }
}

// MARK: - RecognizerViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate

extension RecognizerViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        if let imageSelected = info[UIImagePickerControllerOriginalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = imageSelected

            self.classifyFood(image: imageSelected)
        }

        dismiss(animated: true, completion: nil)
    }

    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
}
