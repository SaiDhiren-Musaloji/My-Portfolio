import React from 'react';
import './App.css';
import { Analytics } from '@vercel/analytics/react';
import Header from './components/Header';
import Hero from './sections/Hero';
import Projects from './sections/Projects';
import Skills from './sections/Skills';
import Career from './sections/Career';
import Contact from './sections/Contact';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Header />
      <main>
        <Hero />
        <Projects />
        <Skills />
        <Career />
        <Contact />
      </main>
      <Footer />
      <Analytics />
    </div>
  );
}

export default App;
