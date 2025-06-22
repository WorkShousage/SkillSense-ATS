import re
from typing import Union, Dict, List, Optional
from PyPDF2 import PdfReader
import docx2txt
import spacy
from datetime import datetime
import json
from spacy.matcher import PhraseMatcher
import logging
from typing import List, Optional, Set, Dict
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")


class ResumeParser:
    def __init__(self):
        # Initialize logging attributes
        self.skill_source_stats = defaultdict(int)
        self.skill_extraction_log = []
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})'
        self.date_pattern = r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b|\b\d{4}\b)'
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            return docx2txt.process(file_path)
        except Exception as e:
            raise Exception(f"Error reading DOCX file: {str(e)}")

    def parse_resume_file(self, file_path: str) -> str:
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        contacts = {}
        emails = re.findall(self.email_pattern, text)
        contacts['email'] = emails[0] if emails else None
        phones = re.finditer(self.phone_pattern, text)
        phone_numbers = [match.group() for match in phones]
        contacts['phone'] = phone_numbers[0] if phone_numbers else None
        return contacts

    def extract_name(self, text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if (line.istitle() and len(line.split()) in (2, 3) and
                    not any(word.lower() in line.lower() for word in
                            ['resume', 'curriculum', 'vitae', 'objective'])):
                return line
        return ""

    def extract_education(self, text: str) -> List[Dict]:
        education = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for i, line in enumerate(lines):
            if 'education' in line.lower():
                for j in range(i + 1, min(i + 6, len(lines))):
                    if any(x in lines[j].lower() for x in ['bachelor', 'master', 'ba', 'bs', 'phd']):
                        institution_line = lines[j - 1] if j > 0 else 'Unknown'
                        # Try to extract year from institution line
                        year_match = re.search(r'(20|19)\d{2}', institution_line)
                        education.append({
                            'institution': re.sub(r'\d{4}', '', institution_line).strip(', '),
                            'degree': lines[j],
                            'year': year_match.group() if year_match else 'Not specified'
                        })
                        break
        return education if education else [{"institution": "Not specified"}]


    def extract_experience_years(self, text: str) -> List[str]:
        import re
        from datetime import datetime
        from dateutil import parser

        pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\s*[–\-—]\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})'
        matches = re.findall(pattern, text, re.IGNORECASE)

        total_months = 0
        now = datetime.now()

        for match in matches:
            try:
                start_str, end_str = re.split(r'\s*[–\-—]\s*', match)
                start = parser.parse(start_str)
                parsed_end = now if 'present' in end_str.lower() else parser.parse(end_str)
                end = min(parsed_end, now)

                months = (end.year - start.year) * 12 + (end.month - start.month)
                total_months += max(months, 0)
            except:
                continue

        if matches:
            years = total_months // 12
            months = total_months % 12
            matches.append(f"Total Experience: {years} years {months} months")
            return matches
        else:
            return ["Not specified"]

    
    def extract_skills(self, text: str, skills_list: Optional[List[str]] = None) -> List[str]:
        """Main skill extraction method with enhanced logging."""
        skills = set()
        self.skill_source_stats.clear()
        self.skill_extraction_log = []
        
        # Log the start of skill extraction
        logger.info("Starting skill extraction process")
        
        # 1. First try to extract from dedicated skills section
        section_skills = self._get_skills_from_section(text)
        skills.update(section_skills)
        self._log_skill_source("Skills Section", section_skills)
        
        # 2. Then try other methods if needed
        if not skills:
            logger.info("No skills found in skills section, trying other methods")
            
            exp_skills = self._get_skills_from_experience(text)
            skills.update(exp_skills)
            self._log_skill_source("Experience Section", exp_skills)
            
            obj_skills = self._get_skills_from_objective(text)
            skills.update(obj_skills)
            self._log_skill_source("Objective Section", obj_skills)
            
            nlp_skills = self._get_skills_via_nlp(text)
            skills.update(nlp_skills)
            self._log_skill_source("NLP Analysis", nlp_skills)
        
        # 3. Use dictionary-based matching if provided
        if skills_list:
            dict_skills = self.extract_skills_with_phrasematcher(text, skills_list)
            skills.update(dict_skills)
            self._log_skill_source("Dictionary Matching", dict_skills)
        
        # Cleanup and return
        filtered_skills = [s.strip().title() for s in skills 
                         if s and len(s.split()) < 5 and len(s) > 2]
        
        # Log final results
        logger.info(f"Skill extraction completed. Found {len(filtered_skills)} skills")
        self._log_skill_stats()
        
        return sorted(filtered_skills) if filtered_skills else ["Not specified"]

    def _log_skill_source(self, source_name: str, skills: List[str]) -> None:
        """Log which skills came from which source."""
        if skills:
            logger.info(f"{source_name} found {len(skills)} skills: {', '.join(skills)}")
            self.skill_source_stats[source_name] += len(skills)
            self.skill_extraction_log.append({
                'source': source_name,
                'skills': skills,
                'count': len(skills)
            })
        else:
            logger.info(f"{source_name} returned no skills")

    def _log_skill_stats(self) -> None:
        """Log statistics about skill sources."""
        if not self.skill_source_stats:
            logger.warning("No skills found from any source")
            return
            
        logger.info("Skill Source Statistics:")
        for source, count in self.skill_source_stats.items():
            logger.info(f"  {source}: {count} skills")
        
        # Log the complete extraction log
        logger.debug("Detailed Skill Extraction Log:")
        for entry in self.skill_extraction_log:
            logger.debug(f"  {entry['source']}: {entry['count']} skills")

    def extract_skills_with_phrasematcher(self, text: str, skills_list: List[str]) -> List[str]:
        """Dictionary-based skill matching with logging."""
        logger.debug(f"Starting dictionary matching with {len(skills_list)} skills")
        
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = list(nlp.pipe(skills_list))
        matcher.add("SKILL", patterns)

        doc = nlp(text)
        matches = matcher(doc)
        found = {doc[start:end].text for match_id, start, end in matches}
        
        logger.debug(f"Dictionary matching found {len(found)} skills")
        return list(found)

    def _get_skills_from_experience(self, text: str) -> List[str]:
        """Extract skills from experience section with detailed logging."""
        logger.debug("Extracting skills from experience section")
        
        section = self._find_section(text, ["experience", "work history", "professional experience"])
        if not section:
            logger.debug("No experience section found")
            return []
        
        skills = set()
        extraction_details = {
            'tool_patterns': set(),
            'action_verbs': set(),
            'bullet_points': set()
        }
        
        # Tool patterns
        tool_patterns = [
            r'(?:using|with|built with|worked with|utilizing|leveraging)\s([A-Za-z0-9+/\- #]+?)(?=[\s,\.;])',
            r'(?:technologies?:|tools?:|stack:)\s*([A-Za-z0-9+/\- ,&]+)'
        ]
        
        for pattern in tool_patterns:
            tools = re.findall(pattern, section, re.IGNORECASE)
            for tool_group in tools:
                extracted = {t.strip().title() for t in re.split(r'[/,&]', tool_group.replace(' and ', ',')) if t.strip() and len(t.strip()) > 2}
                skills.update(extracted)
                extraction_details['tool_patterns'].update(extracted)
        
        # Action verbs
        action_verbs = [
            'developed', 'created', 'implemented', 'designed', 
            'built', 'programmed', 'coded', 'configured',
            'optimized', 'debugged', 'deployed', 'migrated'
        ]
        action_pattern = r'(?:' + '|'.join(action_verbs) + r')\s([\w\s]+?)(?=[\s,\.;])'
        actions = re.findall(action_pattern, section, re.IGNORECASE)
        
        stop_words = {'the', 'a', 'an', 'using', 'with', 'for', 'to', 'in'}
        for action in actions:
            extracted = {w.strip(".,:-").title() for w in action.split() if len(w) > 3 and w.lower() not in stop_words}
            skills.update(extracted)
            extraction_details['action_verbs'].update(extracted)
        
        # Bullet points
        bullet_points = re.findall(r'(?:•|\d+\.|\-)\s*(.*?)(?=\n|$)', section)
        for point in bullet_points:
            if any(verb in point.lower() for verb in action_verbs):
                doc = nlp(point.lower())
                for token in doc:
                    if token.text in action_verbs and token.head.text:
                        skill = token.head.text.title()
                        skills.add(skill)
                        extraction_details['bullet_points'].add(skill)
        
        # Log detailed extraction info
        logger.debug(f"Experience section extraction details:")
        logger.debug(f"  From tool patterns: {extraction_details['tool_patterns']}")
        logger.debug(f"  From action verbs: {extraction_details['action_verbs']}")
        logger.debug(f"  From bullet points: {extraction_details['bullet_points']}")
        
        return list(skills)

    # [Keep all other methods (_get_skills_from_objective, _get_skills_via_nlp, 
    # _get_skills_from_section) with similar logging enhancements]

    def _get_skills_from_objective(self, text: str) -> List[str]:
        """Extract skills from summary/objective section with better NLP."""
        section = self._find_section(text, ["objective", "summary", "profile"])
        if not section:
            return []
        
        skills = set()
        doc = nlp(section)
        
        # Pattern for "X years of experience with Y" 
        exp_pattern = r'(\d+\+?\s*years?.*?experience.*?\bwith\b.*?)(?:\.|,|;|$)'
        exp_matches = re.findall(exp_pattern, section, re.IGNORECASE)
        for match in exp_matches:
            # Extract nouns after "with"
            with_doc = nlp(match.lower())
            for token in with_doc:
                if token.text == 'with' and token.head.text:
                    skills.update(
                        chunk.text.title() 
                        for chunk in with_doc[token.i+1:].noun_chunks
                        if len(chunk.text) > 2
                    )
        
        # Pattern for "proficient in X, Y, Z"
        proficient_pattern = r'(?:proficient|skilled|experienced)\s*(?:in|with)\s*([^\.]+)'
        proficient_matches = re.findall(proficient_pattern, section, re.IGNORECASE)
        for match in proficient_matches:
            skills.update(
                s.strip().title() 
                for s in re.split(r'[,/&]', match)
                if len(s.strip()) > 2
            )
        
        # Extract capitalized phrases that are likely skills
        for sent in doc.sents:
            for ent in sent.ents:
                if ent.label_ in ("ORG", "PRODUCT") and len(ent.text) > 3:
                    skills.add(ent.text.title())
            
            for chunk in sent.noun_chunks:
                if any(t.text in ['experience', 'knowledge'] for t in chunk.root.head.children):
                    skills.add(chunk.text.title())
        
        return list(skills)

    def _get_skills_via_nlp(self, text: str) -> List[str]:
        """Enhanced NLP-based skill extraction using dependency parsing."""
        doc = nlp(text.lower())
        skills = set()
        
        # Define skill indicators
        skill_indicators = {
            'skill', 'experience', 'tool', 'technology', 
            'framework', 'library', 'platform', 'language'
        }
        
        # Extract skills based on dependency relations
        for token in doc:
            # Skills modified by skill indicators
            if token.head.text in skill_indicators and token.dep_ in ('amod', 'compound'):
                skills.add(token.text.title())
            
            # Skills that are objects of skill-related verbs
            if token.head.text in {'use', 'utilize', 'know', 'learn'} and token.dep_ == 'dobj':
                skills.add(token.text.title())
        
        # Extract noun phrases that are likely technologies
        for chunk in doc.noun_chunks:
            # Skip pronouns and very short chunks
            if len(chunk.text) <= 2 or chunk.root.pos_ == 'PRON':
                continue
                
            # Technology-like patterns (e.g., "Python 3", "React.js")
            if (
                re.search(r'[A-Za-z]+[0-9]', chunk.text) or  # Python3
                re.search(r'[A-Za-z]+\.[A-Za-z]+', chunk.text) or  # React.js
                chunk.text in {'java', 'python', 'c++', 'c#', 'sql', 'aws'}  # Common tech
            ):
                skills.add(chunk.text.title())
        
        # Extract from named entities
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "TECH") and 3 <= len(ent.text) <= 30:
                skills.add(ent.text.title())
        
        return list(skills)

    def _get_skills_from_section(self, text: str) -> List[str]:
        section = self._find_section(text, [
            "skills", "technical skills", "key skills", 
            "competencies", "technical competencies",
            "skills & expertise", "skills summary"
        ])
        if not section:
            return []

        skills = set()
        
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Remove bullets or markers
            line = re.sub(r'^[\s•\-*>\d]+', '', line)

            # Handle "skill: level" or "skill (extra)"
            line = re.sub(r'\([^)]*\)', '', line)
            line = line.split(':')[0]

            # Split by common delimiters
            parts = re.split(r'[,/]| and ', line)
            for part in parts:
                skill = part.strip()
                if 2 < len(skill) <= 40 and not skill.isnumeric():
                    skills.add(skill.title())

        return sorted(skills)

    def _find_section(self, text: str, possible_titles: List[str]) -> Optional[str]:
        import logging
        logger = logging.getLogger(__name__)
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            for title in possible_titles:
                title_lower = title.lower()
                # Match only exact or clear start of section header
                if line_lower == title_lower or line_lower.startswith(title_lower + ":"):
                    logger.info(f"🔍 Matched section header: '{line.strip()}' for title: '{title}'")
                    content = []
                    empty_count = 0
                    for next_line in lines[i + 1:]:
                        if any(keyword in next_line.lower() for keyword in ['experience', 'education', 'projects', 'objective']):
                            logger.info(f"⛔ Detected start of next section: '{next_line.strip()}'")
                            break

                        if not next_line.strip():
                            empty_count += 1
                            if empty_count >= 4:  # More lenient: allow sparse formatting
                                logger.info("⛔ Too many blank lines — ending section parse.")
                                break
                            continue

                        empty_count = 0  # Reset count if a real line is found
                        content.append(next_line.strip())

                    section_text = '\n'.join(content).strip()
                    logger.info(f"✅ Extracted content for section '{title}':\n{section_text}\n")
                    return section_text

        # Fallback regex if structured parsing fails
        for title in possible_titles:
            pattern = rf'(?i){re.escape(title)}[:]?(.*?)(?:\n\s*\n|\Z)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                logger.warning(f"⚠️ Used fallback regex to match section: '{title}'")
                return match.group(1).strip()

        logger.warning("❌ No matching section found.")
        return None


    def extract_skills_with_list(self, text: str, skills_list: List[str]) -> List[str]:
        skills = set()
        text_lower = text.lower()
        for skill in skills_list:
            if skill.lower() in text_lower:
                skills.add(skill)
        return sorted(skills) if skills else ["Not specified"]

    def save_to_json(self, data: Dict, output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def parse_resume(self, file_path: str, skills_list: Optional[List[str]] = None) -> Dict:
        try:
            text = self.parse_resume_file(file_path)
            if not text.strip():
                raise ValueError("Empty document or unable to extract text")
            contact_info = self.extract_contact_info(text)
            skills = self.extract_skills_with_list(text, skills_list) if skills_list else self.extract_skills(text)
            return {
                "name": self.extract_name(text),
                "email": contact_info.get('email'),
                "phone": contact_info.get('phone'),
                "skills": skills,
                "education": self.extract_education(text),
                "experience_years": self.extract_experience_years(text),
                "raw_text": text[:1000] + "..." if len(text) > 1000 else text,
                "parsed_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {
                "error": str(e),
                "file": file_path
            }


if __name__ == "__main__":
    parser = ResumeParser()
    resume_data = parser.parse_resume("resumes/sample_resume.docx")
    print("Parsed Resume Data:")
    print(json.dumps(resume_data, indent=2))
    parser.save_to_json(resume_data, "output/parsed_resume.json")
    print("\nSaved to 'parsed_resume.json'")
