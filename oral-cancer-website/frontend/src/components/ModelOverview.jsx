import React from 'react';
import { Layers } from 'lucide-react';

export default function ModelOverview() {
  return (
    <section className="tech-banner" aria-labelledby="model-specs-heading">
      <div className="tech-banner-header">
        <Layers size={18} />
        <h2 id="model-specs-heading" style={{ fontSize: 'inherit', fontWeight: 'inherit', display: 'inline' }}>
          MobileNetV2 Architecture &mdash;
        </h2>
      </div>
      <p>
        This prototype evaluates deep learning oral screening on complete uncropped images. Raw oral photographs are converted to RGB and resized directly to 224&times;224 without lesion localization, bounding boxes, or ROI cropping. A decision threshold of 0.35 is applied to optimize sensitivity for cancer detection.
      </p>

      <div className="tech-specs-grid">
        <div className="spec-item">
          <span className="spec-label">Input Specification</span>
          <span className="spec-value">224 &times; 224 &times; 3 RGB</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">Preprocessing</span>
          <span className="spec-value">Rescaling &times; 1/255 (No ROI)</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">Backbone</span>
          <span className="spec-value">MobileNetV2 (Fine-Tuned)</span>
        </div>
        <div className="spec-item">
          <span className="spec-label">Decision Threshold</span>
          <span className="spec-value">0.35 (Sensitivity-Tuned)</span>
        </div>
      </div>
    </section>
  );
}
