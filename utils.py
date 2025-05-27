import textwrap

categories = textwrap.dedent("""
**sensitive_data**
Contains personal data or special category data. Based on this information, a natural person or 'data subject' can be identified directly or indirectly. This data must be protected in accordance with the General Data Protection Regulation (GDPR).
Examples of sensitive personal data:

- Name
- Address
- ID card or passport number
- Income information
- IP address
- Cultural profile
- Race or ethnic origin
- Sexual orientation or behavior
- Political opinions
- Religious or philosophical beliefs
- Trade union membership
- Genetic data
- Biometric data
- Health data
- Data relating to criminal convictions and offenses
- Other data considered sensitive


**public_data**
Does not contain personal data or special category data. The information does not enable the identification of a natural person either directly or indirectly. This data does not need to be protected under the General Data Protection Regulation (GDPR).
Examples of public data:

- Names of officials who have made public authority decisions in public documents
- Anonymized or pseudonymized names (e.g., "Person T", "Mr. X")
- General demographic data without identifying information
- Contact information of companies or entrepreneurs
- Public registers and openly available information
- General preferences or interests without personal data
- Reviews, opinions, and feedback without personal data
- General statistics and research results
- Company name or registration number
- General email addresses (e.g., info@company.com)
- Other anonymized data
""")
