import React, { useState } from 'react';
import { contactInfo } from '../data/portfolioData';
import SectionLabel from '../components/SectionLabel';
import './Contact.css';

const Contact: React.FC = () => {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [sending, setSending] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSending(true);
    setStatus('idle');
    try {
      const res = await fetch('https://formspree.io/f/xovwrpep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, subject: 'Portfolio contact' }),
      });
      if (res.ok) {
        setStatus('success');
        setFormData({ name: '', email: '', message: '' });
      } else {
        setStatus('error');
      }
    } catch {
      setStatus('error');
    } finally {
      setSending(false);
    }
  };

  return (
    <section id="contact" className="page-section contact-section">
      <div className="contact-section-bg" aria-hidden="true" />
      <div className="page-wrap contact-wrap">
        <SectionLabel
          index="04"
          title="Contact"
          description="Open to roles, collaborations, and hard problems."
          light
        />

        <div className="contact-layout">
          <div className="contact-links-col">
            <a href={`mailto:${contactInfo.email}`} className="contact-big-link">
              {contactInfo.email}
            </a>
            <div className="contact-socials">
              <a href={contactInfo.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn</a>
              <a href={contactInfo.github} target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href={`tel:${contactInfo.phone}`}>{contactInfo.phone}</a>
            </div>
          </div>

          <form className="contact-form" onSubmit={handleSubmit}>
            <input
              type="text"
              name="name"
              placeholder="Name"
              value={formData.name}
              onChange={handleChange}
              required
            />
            <input
              type="email"
              name="email"
              placeholder="Email"
              value={formData.email}
              onChange={handleChange}
              required
            />
            <textarea
              name="message"
              placeholder="Message"
              rows={4}
              value={formData.message}
              onChange={handleChange}
              required
            />
            {status === 'success' && <p className="contact-msg success">Sent — I'll reply soon.</p>}
            {status === 'error' && <p className="contact-msg error">Something went wrong. Try email directly.</p>}
            <button type="submit" disabled={sending}>
              {sending ? 'Sending…' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </section>
  );
};

export default Contact;
