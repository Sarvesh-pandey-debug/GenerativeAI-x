from pydantic import BaseModel, EmailStr, Field
from pydantic import ValidationError

from typing import Optional

class student(BaseModel):
    name: str = "sarvesh"
    age: Optional[int] = None
    email: EmailStr 
    cgpa: Optional[float] = Field(ge=0.0, le=10.0, default=5.0, description="CGPA of the student") # ge = greater than and le - less than 

new_student = {'age': '20', 'email': 'sarvesh@example.com'}


student = student(**new_student)

print(student)


#  we can ad student as dict for particular value fetch 

student_dict = student.dict()
print(student_dict['email'])  # Output: sarvesh@example.com