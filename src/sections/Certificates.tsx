import React, { useState } from 'react';
import { certificates } from '../data/portfolioData';
import './Certificates.css';

const Certificates: React.FC = () => {
  const [selectedCertificate, setSelectedCertificate] = useState<typeof certificates[0] | null>(null);

  const handleCardClick = (certificate: typeof certificates[0]) => {
    setSelectedCertificate(certificate);
  };

  const closeModal = () => {
    setSelectedCertificate(null);
  };

  return (
    <section id="certificates" className="certificates">
      <div className="certificates-container">
        <div className="section-header">
          <h2 className="section-title">Certifications</h2>
          <div className="section-divider"></div>
        </div>

        <div className="certificates-grid">
          {certificates.map(certificate => (
            <div 
              key={certificate.id} 
              className="certificate-card"
              onClick={() => handleCardClick(certificate)}
            >
              <div className="certificate-image">
                <img src={certificate.image} alt={certificate.title} />
                <div className="certificate-overlay">
                  <div className="certificate-actions">
                    <button className="certificate-link view">
                      <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                      </svg>
                      View Certificate
                    </button>
                  </div>
                </div>
              </div>
              
              <div className="certificate-content">
                <h3 className="certificate-title">{certificate.title}</h3>
                <p className="certificate-issuer">{certificate.issuer}</p>
                <p className="certificate-date">Issued: {certificate.issueDate}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Modal for full certificate view */}
        {selectedCertificate && (
          <div className="certificate-modal" onClick={closeModal}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <button className="modal-close" onClick={closeModal}>×</button>
              <img 
                src={selectedCertificate.image} 
                alt={selectedCertificate.title} 
                className="modal-image"
              />
              <div className="modal-info">
                <h3>{selectedCertificate.title}</h3>
                <p>{selectedCertificate.issuer}</p>
                <p>Issued: {selectedCertificate.issueDate}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default Certificates; 