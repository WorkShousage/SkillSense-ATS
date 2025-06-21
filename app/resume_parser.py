import re
from typing import Union, Dict, List, Optional
from PyPDF2 import PdfReader
import docx2txt
import spacy
from datetime import datetime
import json

nlp = spacy.load("en_core_web_sm")


class ResumeParser:
    def __init__(self):
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
                        year_match = re.search(r'(20|19)\d{2}', lines[j + 1] if j + 1 < len(lines) else '')
                        education.append({
                            'institution': lines[j - 1] if j > 0 else 'Unknown',
                            'degree': lines[j],
                            'year': year_match.group() if year_match else 'Not specified'
                        })
                        break
        return education if education else [{"institution": "Not specified"}]

    def extract_experience_years(self, text: str) -> List[str]:
        section = self._find_section(text, ["experience", "work history"])
        if not section:
            return ["Not specified"]
        dates = re.findall(
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(?:20\d{2}|19\d{2})\s*-\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s(?:20\d{2}|19\d{2}))',
            section,
            re.IGNORECASE
        )
        return dates if dates else ["Not specified"]

    def extract_skills(self, text: str) -> List[str]:
        skills = set()
        skills.update(self._get_skills_from_section(text))
        skills.update(self._get_skills_from_experience(text))
        skills.update(self._get_skills_from_objective(text))
        skills.update(self._get_skills_via_nlp(text))
        filtered_skills = [s for s in skills if s and len(s.split()) < 4]
        return sorted(filtered_skills) if filtered_skills else ["Not specified"]

    def _get_skills_from_section(self, text: str) -> List[str]:
        section = self._find_section(text, ["skills", "technical skills", "competencies"])
        if not section:
            return []
        skills = set()
        skills.update(re.findall(r'[•\-*]\s*([^\n]+)', section))
        skills.update(line.strip() for line in section.split('\n') if line.strip())
        return [s.title() for s in skills]

    def _get_skills_from_experience(self, text: str) -> List[str]:
        section = self._find_section(text, ["experience", "work history"])
        if not section:
            return []
        skills = set()
        tools = re.findall(r'(?:using|with|built with|worked with)\s([A-Za-z0-9+/\- ]+?)(?=[\s,\.])', section, re.IGNORECASE)
        for tool_group in tools:
            skills.update(t.strip().title() for t in re.split(r'[/,&]', tool_group))
        actions = re.findall(r'(?:designed|developed|created|implemented)\s([\w\s]+?)(?=[\s,\.])', section, re.IGNORECASE)
        for action in actions:
            skills.update(w.strip().title() for w in action.split() if len(w) > 3)
        return list(skills)

    def _get_skills_from_objective(self, text: str) -> List[str]:
        section = self._find_section(text, ["objective", "summary"])
        if not section:
            return []
        skills = set()
        keywords = ["skills", "experience", "proficient", "knowledge"]
        for line in section.split('\n'):
            if any(kw in line.lower() for kw in keywords):
                skills.update(w.strip(".,").title() for w in line.split() if w[0].isupper())
        return list(skills)

    def _get_skills_via_nlp(self, text: str) -> List[str]:
        doc = nlp(text.lower())
        skills = set()
        for chunk in doc.noun_chunks:
            if any(token.text in ["skill", "experience", "tool"] for token in chunk.root.head.children):
                skills.add(chunk.text.title())
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT") and len(ent.text) > 3:
                skills.add(ent.text.title())
        return list(skills)

    def _find_section(self, text: str, possible_titles: List[str]) -> Optional[str]:
        for title in possible_titles:
            pattern = rf'(?i){re.escape(title)}[:]?(.+?)(?:\n\n|\Z)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
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
