import React, { useState, useEffect } from 'react';
import './Header.css';

const NAV = [
  { id: 'projects', label: 'Work' },
  { id: 'skills', label: 'Stack' },
  { id: 'career', label: 'Career' },
  { id: 'contact', label: 'Contact' },
];

const Header: React.FC = () => {
  const [onHero, setOnHero] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => {
      setOnHero(window.scrollY < window.innerHeight * 0.6);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setMenuOpen(false);
  };

  return (
    <header className={`site-header ${onHero ? 'on-hero' : ''}`}>
      <div className="site-header-pill">
        <button type="button" className="site-brand" onClick={() => go('hero')}>
          SD
        </button>

        <nav className={`site-nav ${menuOpen ? 'open' : ''}`}>
          {NAV.map((item) => (
            <button key={item.id} type="button" onClick={() => go(item.id)}>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="site-header-end">
          <a href="/resume.pdf" className="site-resume" target="_blank" rel="noopener noreferrer">
            Resume
          </a>
          <button
            type="button"
            className={`site-menu-btn ${menuOpen ? 'open' : ''}`}
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Menu"
          >
            <span /><span /><span />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
