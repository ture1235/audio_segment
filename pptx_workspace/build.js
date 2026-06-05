const pptxgen = require('pptxgenjs');
const html2pptx = require('/home/dutrue/.claude/skills/pptx/scripts/html2pptx');
const sharp = require('sharp');
const path = require('path');

async function createGradient(filename, color1, color2) {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="562.5">
    <defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#${color1}"/>
      <stop offset="100%" style="stop-color:#${color2}"/>
    </linearGradient></defs>
    <rect width="100%" height="100%" fill="url(#g)"/>
  </svg>`;
  await sharp(Buffer.from(svg)).png().toFile(filename);
  return filename;
}

async function main() {
  // Rasterize backgrounds
  await createGradient('slides/title-bg.png', '1B2A4A', '0F1B33');
  await createGradient('slides/section-bg.png', '1B2A4A', '2EC4B6');

  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Audio Segment Project';
  pptx.title = 'Alarm Sound Classification System';

  const slideDir = path.join(__dirname, 'slides');

  // Slide 1: Title
  await html2pptx(path.join(slideDir, '01_title.html'), pptx);

  // Slide 2: Project Overview
  await html2pptx(path.join(slideDir, '02_overview.html'), pptx);

  // Slide 3: Alarm Types
  await html2pptx(path.join(slideDir, '03_alarms.html'), pptx);

  // Slide 4: Core Challenges
  await html2pptx(path.join(slideDir, '04_challenges.html'), pptx);

  // Slide 5: Data Pipeline
  await html2pptx(path.join(slideDir, '05_data_pipeline.html'), pptx);

  // Slide 6: Model Architecture
  await html2pptx(path.join(slideDir, '06_architecture.html'), pptx);

  // Slide 7: Architecture Details
  await html2pptx(path.join(slideDir, '07_arch_detail.html'), pptx);

  // Slide 8: Training Strategy
  await html2pptx(path.join(slideDir, '08_training.html'), pptx);

  // Slide 9: Inference Pipeline
  await html2pptx(path.join(slideDir, '09_inference.html'), pptx);

  // Slide 10: Optimization Journey
  await html2pptx(path.join(slideDir, '10_optimization.html'), pptx);

  // Slide 11: Results & Summary
  await html2pptx(path.join(slideDir, '11_results.html'), pptx);

  await pptx.writeFile({ fileName: 'Alarm_Sound_Classification_Project.pptx' });
  console.log('Done!');
}

main().catch(e => { console.error(e); process.exit(1); });
