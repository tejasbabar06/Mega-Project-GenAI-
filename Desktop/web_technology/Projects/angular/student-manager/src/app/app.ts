import { Component } from '@angular/core';
import { StudentInputComponent, Student } from './components/student-input/student-input.component';
import { StudentListComponent } from './components/student-list/student-list.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [StudentInputComponent, StudentListComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  // Application title
  title = 'Student Manager Lite';

  // Array to store all students
  students: Student[] = [];

  // Counter to generate unique ids for each student
  private nextId = 1;

  /**
   * Adds a new student to the list.
   * Called when StudentInputComponent emits the studentAdded event.
   */
  onStudentAdded(data: { name: string; age: number }): void {
    const newStudent: Student = {
      id: this.nextId++,  // Assign unique id and increment counter
      name: data.name,
      age: data.age
    };
    this.students.push(newStudent);
  }

  /**
   * Deletes a student by their unique id.
   * Called when StudentListComponent emits the studentDeleted event.
   */
  onStudentDeleted(id: number): void {
    // Filter out the student with the matching id
    this.students = this.students.filter(student => student.id !== id);
  }
}
