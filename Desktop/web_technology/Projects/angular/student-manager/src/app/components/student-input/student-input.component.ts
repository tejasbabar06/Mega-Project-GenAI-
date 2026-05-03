import { Component, Output, EventEmitter } from '@angular/core';
import { FormsModule } from '@angular/forms';

// Interface defining the structure of a Student object
export interface Student {
  id: number;
  name: string;
  age: number;
}

@Component({
  selector: 'app-student-input',
  standalone: true,
  imports: [FormsModule], // FormsModule is required for [(ngModel)]
  templateUrl: './student-input.component.html',
  styleUrl: './student-input.component.css'
})
export class StudentInputComponent {
  // Two-way bound properties for the input fields
  studentName: string = '';
  studentAge: number | null = null;

  // EventEmitter to send new student data to the parent component
  @Output() studentAdded = new EventEmitter<{ name: string; age: number }>();

  /**
   * Handles the "Add Student" button click.
   * Validates inputs before emitting the student data.
   */
  onAddStudent(): void {
    // Trim whitespace and check for empty name
    const trimmedName = this.studentName.trim();

    // Do not allow empty name input
    if (!trimmedName) {
      alert('Please enter a student name.');
      return;
    }

    // Validate age is a positive number
    if (!this.studentAge || this.studentAge <= 0) {
      alert('Please enter a valid age.');
      return;
    }

    // Emit the new student data to the parent
    this.studentAdded.emit({
      name: trimmedName,
      age: this.studentAge
    });

    // Clear input fields after adding student
    this.studentName = '';
    this.studentAge = null;
  }
}
